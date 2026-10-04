//! actual-CLI KV certification cases.
use super::{fixture, process};
#[path = "kv_peer.rs"]
mod peer;
use peer::{Behavior, PIN, PRIMARY, SECONDARY, Server};
use serde_json::Value;
use std::{fs, path::Path, sync::atomic::Ordering};
fn args(base: &str, output: &Path) -> Vec<String> {
    [
        "automation",
        "stability",
        "kv-tool-loop",
        "--base-url",
        base,
        "--models",
        "fixture",
        "--attempts",
        "2",
        "--pressure-turns",
        "2",
        "--overlap-requests",
        "4",
        "--timeout",
        "2",
        "--min-cached-tokens",
        "2048",
        "--suffix-prefill-limit",
        "256",
        "--output-dir",
        output.to_str().unwrap(),
    ]
    .map(str::to_owned)
    .to_vec()
}
fn invoke(cwd: &Path, args: &[String]) -> process::RawProcessReport {
    fixture::invoke_with(
        cwd,
        args,
        std::collections::BTreeMap::from([(
            "MESH_KV_TOOL_LOOP_NATIVE_LOGS".into(),
            process::Value::Public("".into()),
        )]),
    )
}
fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
fn last_prompt(payload: &Value) -> &str {
    payload["messages"].as_array().unwrap().last().unwrap()["content"]
        .as_str()
        .unwrap_or("")
}
#[test]
fn stability_cli_kv_full_pressure_and_each_attempt_overlap_preserve_history_cache_and_fresh_transcripts()
 {
    let server = Server::new(4, Behavior::Healthy);
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("evidence");
    fs::create_dir_all(output.join("transcripts")).unwrap();
    fs::write(output.join("transcripts/foreign.jsonl"), b"old evidence").unwrap();
    let result = invoke(directory.path(), &args(&server.base, &output));
    assert_eq!(result.process.status.unwrap().code(), Some(0));
    let summary = read(&output.join("summary.json"));
    assert_eq!(summary["ok"], true);
    assert_eq!(summary["total"], 6);
    assert_eq!(summary["passed"], 6);
    assert_eq!(summary["phases"]["tool_loop"]["total"], 2);
    assert_eq!(summary["phases"]["overlap_tool_loop"]["total"], 2);
    assert_eq!(server.cohorts.load(Ordering::SeqCst), 2);
    let requests = server.requests.lock().unwrap().clone();
    assert_eq!(requests.len(), 46);
    assert_eq!(
        requests
            .iter()
            .filter(|request| last_prompt(request).starts_with("Pressure turn "))
            .count(),
        4
    );
    let recalls = requests
        .iter()
        .filter(|request| last_prompt(request).starts_with("Final recall:"))
        .collect::<Vec<_>>();
    assert_eq!(recalls.len(), 8);
    for request in &requests {
        assert_eq!(request["stream"], false);
        assert_eq!(request["reasoning_effort"], "none");
        assert_eq!(request["chat_template_kwargs"]["enable_thinking"], false);
        let text = request.to_string();
        assert!(!text.contains("private fixture reasoning"));
        assert!(!text.contains("private_marker"));
    }
    for recall in recalls {
        let messages = recall["messages"].as_array().unwrap();
        let tools = messages
            .iter()
            .filter(|message| message["role"] == "tool")
            .collect::<Vec<_>>();
        assert_eq!(tools.len(), 2);
        let facts = tools
            .iter()
            .map(|tool| serde_json::from_str::<Value>(tool["content"].as_str().unwrap()).unwrap())
            .collect::<Vec<_>>();
        assert!(facts.iter().any(|fact| fact["value"] == PRIMARY));
        assert!(facts.iter().any(|fact| fact["value"] == SECONDARY));
        for tool in tools {
            assert!(messages.iter().any(|message| message["role"] == "assistant"
                && message["tool_calls"][0]["id"] == tool["tool_call_id"]));
        }
    }
    let tail = &requests[requests.len() - 4..];
    assert_eq!(tail[0]["messages"][0], tail[1]["messages"][0]);
    assert_ne!(tail[0], tail[1]);
    assert_eq!(tail[2], tail[3]);
    let rows = fixture::rows(&output.join("results.jsonl"));
    for row in rows.iter().filter(|row| row["phase"] != "tool_loop") {
        assert_eq!(row["prompt_tokens"], 2304);
        assert_eq!(row["cached_tokens"], 2048);
    }
    let manifest = read(&output.join("manifest.json"));
    let entries = manifest["transcript_files"].as_array().unwrap();
    assert_eq!(entries.len(), 10);
    for entry in entries {
        let path = entry["path"].as_str().unwrap();
        let selected = Path::new(path);
        assert!(selected.starts_with("transcripts"));
        assert!(
            selected
                .parent()
                .unwrap()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("run-")
        );
        assert!(output.join(path).is_file());
        assert!(!path.contains("foreign"));
    }
    assert_eq!(
        fs::read(output.join("transcripts/foreign.jsonl")).unwrap(),
        b"old evidence"
    );
    let first = entries[0]["path"].as_str().unwrap().to_owned();
    let second = invoke(directory.path(), &args(&server.base, &output));
    assert_eq!(second.process.status.unwrap().code(), Some(0));
    let current = read(&output.join("manifest.json"));
    assert_ne!(current["transcript_files"][0]["path"], first);
    assert!(output.join(first).is_file());
    assert_eq!(server.cohorts.load(Ordering::SeqCst), 4);
}
#[test]
fn stability_cli_kv_failed_overlap_sibling_retains_successful_histories_and_measured_cache_evidence()
 {
    let server = Server::new(4, Behavior::FailedOverlap);
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("evidence");
    let result = invoke(directory.path(), &args(&server.base, &output));
    assert_eq!(result.process.status.unwrap().code(), Some(1));
    let summary = read(&output.join("summary.json"));
    assert_eq!(summary["ok"], false);
    assert_eq!(summary["total"], 6);
    assert_eq!(summary["failed"], 1);
    assert_eq!(server.cohorts.load(Ordering::SeqCst), 2);
    let rows = fixture::rows(&output.join("results.jsonl"));
    let failed = rows.iter().find(|row| row["ok"] == false).unwrap();
    assert_eq!(failed["phase"], "overlap_tool_loop");
    assert_eq!(failed["cached_tokens"], 2048);
    assert!(failed["detail"].as_str().unwrap().contains("503"));
    let manifest = read(&output.join("manifest.json"));
    let entries = manifest["transcript_files"].as_array().unwrap();
    let mut failed_statuses = 0;
    let mut successful_final = 0;
    for entry in entries.iter().filter(|entry| {
        entry["cohort"].as_str().unwrap().starts_with("overlap_") && entry["attempt"] == 1
    }) {
        for record in fixture::rows(&output.join(entry["path"].as_str().unwrap())) {
            if record["status_code"] == 503 {
                failed_statuses += 1;
            }
            if record["phase"] == "final_recall" && record["error"].is_null() {
                successful_final += 1;
            }
        }
    }
    assert_eq!(failed_statuses, 1);
    assert_eq!(successful_final, 2);
}
#[test]
fn stability_cli_kv_wrong_pressure_or_cached_completion_fails_with_observations_retained() {
    for behavior in [Behavior::WrongPressure, Behavior::WrongCache] {
        let server = Server::new(4, behavior);
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("evidence");
        let result = invoke(directory.path(), &args(&server.base, &output));
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        let summary = read(&output.join("summary.json"));
        assert_eq!(summary["ok"], false);
        let rows = fixture::rows(&output.join("results.jsonl"));
        match behavior {
            Behavior::WrongPressure => {
                assert_eq!(summary["phases"]["tool_loop"]["failed"], 2);
                assert_eq!(summary["phases"]["overlap_tool_loop"]["passed"], 2);
                assert!(
                    !server
                        .requests
                        .lock()
                        .unwrap()
                        .iter()
                        .any(|request| last_prompt(request).starts_with("Pressure turn 2"))
                );
            }
            Behavior::WrongCache => {
                let failed = rows
                    .iter()
                    .filter(|row| row["ok"] == false)
                    .collect::<Vec<_>>();
                assert_eq!(failed.len(), 3);
                for row in failed {
                    assert_eq!(row["cached_tokens"], 2048);
                    assert_eq!(row["prompt_tokens"], 2304);
                    assert!(row["detail"].as_str().unwrap().contains(PIN));
                }
            }
            _ => unreachable!(),
        }
    }
}
#[test]
fn stability_cli_kv_plan_and_scoped_help_do_not_create_evidence_or_contact_endpoint() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("not-created");
    let mut plan_args = args("http://127.0.0.1:1/tenant", &output);
    plan_args.push("--print-plan".into());
    let first = invoke(directory.path(), &plan_args);
    let second = invoke(directory.path(), &plan_args);
    assert_eq!(first.process.status.unwrap().code(), Some(0));
    assert_eq!(
        first.stdout.as_ref().unwrap().as_bytes(),
        second.stdout.as_ref().unwrap().as_bytes()
    );
    let plan: Value = serde_json::from_slice(first.stdout.unwrap().as_bytes()).unwrap();
    let checks = plan["checks"].as_array().unwrap();
    assert_eq!(checks.len(), 6);
    assert_eq!(
        checks
            .iter()
            .filter(|check| check["phase"] == "overlap_tool_loop")
            .count(),
        2
    );
    assert!(!output.exists());
    let help = invoke(
        directory.path(),
        &["automation", "stability", "kv-tool-loop", "--help"].map(str::to_owned),
    );
    assert_eq!(help.process.status.unwrap().code(), Some(0));
    assert!(!output.exists());
}

#[test]
fn stability_cli_kv_native_logs_checkpoint_before_requests_and_detect_new_fatal_lines() {
    for preexisting in [true, false] {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("evidence");
        let log = directory.path().join("native log with spaces.txt");
        let old = "llama_decode failed before test\nhealthy old\n";
        if preexisting {
            fs::write(&log, old).unwrap();
        }
        let server = Server::with_log(4, Behavior::Healthy, Some(log));
        let mut inputs = args(&server.base, &output);
        inputs.extend(["--native-log", "native log with spaces.txt"].map(str::to_owned));
        let result = invoke(directory.path(), &inputs);
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        let summary = read(&output.join("summary.json"));
        assert_eq!(summary["total"], 7);
        assert_eq!(summary["failed"], 1);
        assert_eq!(summary["phases"]["native_log_scan"]["failed"], 1);
        assert_eq!(server.cohorts.load(Ordering::SeqCst), 2);
        let scans = read(&output.join("native-log-scan.json"));
        let scan = &scans[0];
        assert_eq!(scan["total_findings"], 1);
        assert_eq!(scan["rescanned"], !preexisting);
        assert_eq!(
            scan["start_offset"],
            if preexisting { old.len() } else { 0 }
        );
        assert_eq!(
            scan["findings"][0]["line_number"],
            if preexisting { 3 } else { 1 }
        );
        assert!(
            scan["findings"][0]["text"]
                .as_str()
                .unwrap()
                .contains("after request")
        );
    }
}
#[test]
fn stability_cli_kv_failed_native_log_checkpoint_prevents_all_endpoint_requests() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("evidence");
    let invalid = directory.path().join("log directory");
    fs::create_dir(&invalid).unwrap();
    let server = Server::new(4, Behavior::Healthy);
    let mut inputs = args(&server.base, &output);
    inputs.extend(["--native-log".into(), invalid.to_str().unwrap().into()]);
    let result = invoke(directory.path(), &inputs);
    assert_eq!(result.process.status.unwrap().code(), Some(1));
    assert!(server.requests.lock().unwrap().is_empty());
    let summary = read(&output.join("summary.json"));
    assert_eq!(summary["ok"], false);
    assert_eq!(summary["phases"]["native_log_scan"]["failed"], 1);
    assert!(
        fixture::rows(&output.join("results.jsonl"))[0]["detail"]
            .as_str()
            .unwrap()
            .contains("regular file")
    );
    assert_eq!(fs::read_dir(invalid).unwrap().count(), 0);
}
#[cfg(unix)]
#[test]
fn stability_cli_kv_interrupt_closes_overlapping_connections_and_retains_partial_evidence() {
    use std::{
        thread,
        time::{Duration, Instant},
    };
    let server = Server::new(4, Behavior::HeldOverlap);
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("evidence");
    let mut inputs = args(&server.base, &output);
    inputs.extend(["--timeout", "10"].map(str::to_owned));
    let cancellation = process::Cancellation::default();
    let cancel = cancellation.clone();
    let cohorts = server.cohorts.clone();
    let interrupt = thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(3);
        while cohorts.load(Ordering::SeqCst) == 0 {
            if Instant::now() >= deadline {
                return false;
            }
            thread::sleep(Duration::from_millis(5));
        }
        cancel.cancel();
        true
    });
    let environment = std::collections::BTreeMap::from([(
        "MESH_KV_TOOL_LOOP_NATIVE_LOGS".into(),
        process::Value::Public("".into()),
    )]);
    let result =
        fixture::invoke_with_cancellation(directory.path(), &inputs, environment, &cancellation);
    assert!(interrupt.join().unwrap());
    assert_eq!(result.process.status.unwrap().code(), Some(143));
    assert!(result.process.cleanup.complete && !result.process.cleanup.forced);
    let summary = read(&output.join("summary.json"));
    assert_eq!(summary["ok"], false);
    assert_eq!(summary["cancelled"], true);
    assert_eq!(summary["total"], 2);
    assert_eq!(server.requests.lock().unwrap().len(), 10);
    let deadline = Instant::now() + Duration::from_secs(2);
    while server.closed.load(Ordering::SeqCst) < 4 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(
        server.closed.load(Ordering::SeqCst),
        4,
        "owned overlap connections did not close"
    );
    let manifest = read(&output.join("manifest.json"));
    assert_eq!(manifest["transcript_files"].as_array().unwrap().len(), 5);
    for entry in manifest["transcript_files"].as_array().unwrap() {
        assert!(output.join(entry["path"].as_str().unwrap()).is_file());
    }
}

#[test]
fn stability_cli_kv_wrong_tool_identity_stops_that_history_and_retains_the_valid_cohort() {
    for behavior in [Behavior::WrongToolKey, Behavior::MalformedTool] {
        let server = Server::new(4, behavior);
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("evidence");
        let result = invoke(directory.path(), &args(&server.base, &output));
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        let summary = read(&output.join("summary.json"));
        assert_eq!(summary["total"], 6);
        assert_eq!(summary["failed"], 2);
        assert_eq!(summary["phases"]["tool_loop"]["failed"], 2);
        assert_eq!(summary["phases"]["overlap_tool_loop"]["passed"], 2);
        for request in server.requests.lock().unwrap().iter() {
            if request["messages"][1]["content"]
                .as_str()
                .unwrap()
                .starts_with("Attempt ")
            {
                assert_eq!(
                    request["messages"].as_array().unwrap().len(),
                    2,
                    "invalid tool response was forwarded into history"
                );
            }
        }
        for row in fixture::rows(&output.join("results.jsonl"))
            .iter()
            .filter(|row| row["ok"] == false)
        {
            assert_eq!(row["status_code"], 200);
            assert_eq!(row["phase"], "tool_loop");
            assert!(row["detail"].as_str().unwrap().contains("KV tool"));
        }
    }
}
#[test]
fn stability_cli_kv_cache_shortfall_fails_each_proof_with_measured_tokens_retained() {
    for behavior in [Behavior::CacheShortfall, Behavior::SuffixShortfall] {
        let server = Server::new(4, behavior);
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("evidence");
        let result = invoke(directory.path(), &args(&server.base, &output));
        assert_eq!(result.process.status.unwrap().code(), Some(1));
        let summary = read(&output.join("summary.json"));
        assert_eq!(summary["total"], 6);
        assert_eq!(summary["failed"], 4);
        assert_eq!(summary["phases"]["tool_loop"]["passed"], 2);
        let (prompt, cached, detail) = if matches!(behavior, Behavior::CacheShortfall) {
            (2304, 2047, "cached_tokens=2047")
        } else {
            (2305, 2048, "suffix_prefill_tokens=257")
        };
        for row in fixture::rows(&output.join("results.jsonl"))
            .iter()
            .filter(|row| row["ok"] == false)
        {
            assert_eq!(row["prompt_tokens"], prompt);
            assert_eq!(row["cached_tokens"], cached);
            assert_eq!(row["status_code"], 200);
            assert!(row["detail"].as_str().unwrap().contains(detail));
        }
    }
}
