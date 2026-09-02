use super::*;
use mesh_llm_events::logging::{
    envelope::{CanonicalEnvelope, CanonicalPresentationContext},
    events::{LifecycleEvent, TokenUsage},
    identifiers::{EventId, RequestId},
    replay::ReplayChannel,
};
use std::process::Command;

const TERMINAL_BOUNDARY_CHILD_MODE_ENV: &str = "MESH_LLM_TUI_TERMINAL_BOUNDARY_CHILD_MODE";
const TERMINAL_BOUNDARY_CHILD_TEST: &str =
    "output::tests::logging_projection::canonical_lifecycle_terminal_boundary_child";
const TERMINAL_BOUNDARY_PRIVATE_MARKERS: [&str; 6] = [
    "terminal-boundary-bearer-secret",
    "terminal-boundary-prompt",
    "terminal-boundary-tenant",
    "terminal-boundary-account",
    "terminal-boundary-user",
    "terminal-boundary-role",
];

fn canonical_output(request_id: RequestId, sequence: u64, event: LifecycleEvent) -> OutputEvent {
    OutputEvent::CanonicalLog(Box::new(CanonicalEnvelope::new(
        EventId::new(),
        request_id,
        ReplayChannel::Requests,
        sequence,
        "2026-08-04T12:34:56.789Z".to_string(),
        event,
    )))
}

fn canonical_terminal_boundary_output() -> OutputEvent {
    OutputEvent::CanonicalLog(Box::new(
        CanonicalEnvelope::new(
            EventId::new(),
            RequestId::new(),
            ReplayChannel::Requests,
            17,
            "2026-08-04T12:34:56.789Z".to_string(),
            LifecycleEvent::Failed {
                error: "Bearer terminal-boundary-bearer-secret prompt=terminal-boundary-prompt"
                    .to_string(),
                status_code: None,
            },
        )
        .with_tenant("terminal-boundary-tenant".to_string())
        .with_account("terminal-boundary-account".to_string())
        .with_user("terminal-boundary-user".to_string())
        .with_role("terminal-boundary-role".to_string()),
    ))
}

fn run_terminal_boundary_child(mode: &str) -> std::process::Output {
    Command::new(std::env::current_exe().expect("resolve current TUI test executable"))
        .args(["--exact", TERMINAL_BOUNDARY_CHILD_TEST, "--nocapture"])
        .env(TERMINAL_BOUNDARY_CHILD_MODE_ENV, mode)
        .output()
        .expect("run terminal-boundary child process")
}

fn assert_terminal_stream_is_private(stream_name: &str, stream: &str) {
    for private_marker in TERMINAL_BOUNDARY_PRIVATE_MARKERS {
        assert!(
            !stream.contains(private_marker),
            "{stream_name} leaked {private_marker:?}: {stream:?}"
        );
    }
}

#[test]
fn canonical_jsonl_is_one_stable_safe_record_per_line() {
    let request_id = RequestId::new();
    let event = canonical_output(
        request_id,
        7,
        LifecycleEvent::Completed {
            status_code: Some(200),
            duration_ms: Some(31),
            usage: Some(TokenUsage {
                prompt_tokens: Some(8),
                cached_prompt_tokens: None,
                completion_tokens: Some(3),
                total_tokens: Some(11),
            }),
        },
    );

    let rendered = JsonFormatter
        .format(&event)
        .expect("canonical event should format as JSONL");
    assert_eq!(rendered.lines().count(), 1);
    let value: Value = serde_json::from_str(rendered.trim_end()).expect("valid JSONL record");

    assert_eq!(value["timestamp"], "2026-08-04T12:34:56.789Z");
    assert_eq!(value["level"], "info");
    assert_eq!(value["event"], "request_completed");
    assert_eq!(value["channel"], "requests");
    assert_eq!(value["request_kind"], "unknown");
    assert_eq!(value["source"], "unknown");
    assert_eq!(value["sequence"], 7);
    assert_eq!(value["outcome"], "completed");
    assert_eq!(value["status_code"], 200);
    assert_eq!(value["duration_ms"], 31);
    assert_eq!(value["prompt_tokens"], 8);
    assert_eq!(value["completion_tokens"], 3);
    assert_eq!(value["total_tokens"], 11);
    assert_eq!(value["request_id"], request_id.as_uuid().to_string());
    assert!(value["event_id"].as_str().is_some());
    assert!(value.get("tokens").is_none());
    assert!(rendered.contains(&request_id.as_uuid().to_string()));
}

#[test]
fn canonical_projection_exposes_truthful_request_context_in_pretty_and_jsonl() {
    let request_id = RequestId::new();
    let event = OutputEvent::CanonicalLog(Box::new(
        CanonicalEnvelope::new(
            EventId::new(),
            request_id,
            ReplayChannel::Requests,
            12,
            "2026-08-04T12:34:56.789Z".to_string(),
            LifecycleEvent::Completed {
                status_code: Some(200),
                duration_ms: Some(18),
                usage: None,
            },
        )
        .with_presentation_context(CanonicalPresentationContext::from_parts(
            Some("chat_completions"),
            Some("mesh_forwarded"),
            Some("safe-model"),
            Some("mesh-provider"),
            Some("raw-engine"),
            Some("POST"),
        )),
    ));

    let json_line = JsonFormatter.format(&event).expect("canonical JSONL");
    let record: Value = serde_json::from_str(json_line.trim_end()).expect("JSONL record");
    assert_eq!(record["request_kind"], "inference");
    assert_eq!(record["route"], "chat_completions");
    assert_eq!(record["source"], "mesh_forwarded");
    assert_eq!(record["model"], "safe-model");
    assert_eq!(record["provider"], "mesh-provider");
    assert_eq!(record["engine"], "raw-engine");
    assert_eq!(record["method"], "POST");
    assert_eq!(record["status_code"], 200);
    assert_eq!(record["duration_ms"], 18);

    let pretty = PrettyFormatter.format(&event).expect("canonical pretty");
    assert!(
        pretty.contains("inference request completed route=chat_completions source=mesh_forwarded")
    );
    assert!(
        pretty.contains("model=safe-model provider=mesh-provider engine=raw-engine method=POST"),
        "pretty={pretty:?}"
    );
    assert!(pretty.contains("status=200 duration=18ms"));
    assert!(pretty.contains(&request_id.as_uuid().to_string()));
}

#[test]
fn canonical_jsonl_uses_stdout_and_pretty_uses_stderr() {
    let event = canonical_output(
        RequestId::new(),
        1,
        LifecycleEvent::Admitted {
            model: Some("private-model-name".to_string()),
            method: Some("POST".to_string()),
        },
    );
    let mut json = JsonFormatter;
    let mut pretty = PrettyFormatter;
    let json_line = json.format(&event).expect("json line");
    let pretty_line = pretty.format(&event).expect("pretty line");
    let mut json_stdout = Vec::new();
    let mut json_stderr = Vec::new();
    let mut pretty_stdout = Vec::new();
    let mut pretty_stderr = Vec::new();

    write_rendered_output_to_writers(
        LogFormat::Json,
        &json_line,
        &mut json_stdout,
        &mut json_stderr,
    )
    .expect("JSON output routing");
    write_rendered_output_to_writers(
        LogFormat::Pretty,
        &pretty_line,
        &mut pretty_stdout,
        &mut pretty_stderr,
    )
    .expect("pretty output routing");

    assert!(!json_stdout.is_empty());
    assert!(json_stderr.is_empty());
    assert!(pretty_stdout.is_empty());
    assert!(!pretty_stderr.is_empty());
}

#[test]
fn canonical_lifecycle_process_boundary_keeps_jsonl_stdout_and_pretty_tui_stderr_private() {
    let json_output = run_terminal_boundary_child("json");
    let pretty_tui_output = run_terminal_boundary_child("pretty_tui");

    assert!(
        json_output.status.success(),
        "JSON child failed: stdout={:?}, stderr={:?}",
        String::from_utf8_lossy(&json_output.stdout),
        String::from_utf8_lossy(&json_output.stderr),
    );
    assert!(
        pretty_tui_output.status.success(),
        "pretty/TUI child failed: stdout={:?}, stderr={:?}",
        String::from_utf8_lossy(&pretty_tui_output.stdout),
        String::from_utf8_lossy(&pretty_tui_output.stderr),
    );

    let json_stdout = String::from_utf8(json_output.stdout).expect("JSON child stdout is UTF-8");
    let json_stderr = String::from_utf8(json_output.stderr).expect("JSON child stderr is UTF-8");
    let pretty_tui_stdout =
        String::from_utf8(pretty_tui_output.stdout).expect("pretty/TUI child stdout is UTF-8");
    let pretty_tui_stderr =
        String::from_utf8(pretty_tui_output.stderr).expect("pretty/TUI child stderr is UTF-8");

    let json_records = json_stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|record| record["event"] == "request_failed")
        .collect::<Vec<_>>();
    assert_eq!(
        json_records.len(),
        1,
        "the child must emit exactly one canonical JSONL lifecycle record: {json_stdout:?}"
    );
    let json_record = &json_records[0];
    assert_eq!(json_record["event"], "request_failed");
    assert_eq!(json_record["channel"], "requests");
    assert_eq!(json_record["sequence"], 17);
    assert!(json_record["request_id"].as_str().is_some());
    assert!(json_record["event_id"].as_str().is_some());
    assert!(
        !json_stderr.contains("request_failed"),
        "canonical JSONL must not be written to stderr: {json_stderr:?}"
    );

    assert!(
        !pretty_tui_stdout.contains("request failed"),
        "pretty/TUI lifecycle output must not be written to stdout: {pretty_tui_stdout:?}"
    );
    assert!(
        pretty_tui_stderr.contains("request failed"),
        "pretty/TUI lifecycle output must reach stderr: {pretty_tui_stderr:?}"
    );
    assert!(pretty_tui_stderr.contains("request_id="));

    for (stream_name, stream) in [
        ("JSON stdout", json_stdout.as_str()),
        ("JSON stderr", json_stderr.as_str()),
        ("pretty/TUI stdout", pretty_tui_stdout.as_str()),
        ("pretty/TUI stderr", pretty_tui_stderr.as_str()),
    ] {
        assert_terminal_stream_is_private(stream_name, stream);
    }
}

/// Process fixture for the parent boundary test above. The interactive selection
/// acts as a pseudo-TTY: before the alternate screen is entered, it emits the
/// same pretty event line that a TUI session sends to the terminal boundary.
#[test]
fn canonical_lifecycle_terminal_boundary_child() {
    let Ok(mode) = std::env::var(TERMINAL_BOUNDARY_CHILD_MODE_ENV) else {
        return;
    };
    let event = canonical_terminal_boundary_output();

    let (format, rendered) = match mode.as_str() {
        "json" => (
            LogFormat::Json,
            JsonFormatter
                .format(&event)
                .expect("format canonical JSONL lifecycle event"),
        ),
        "pretty_tui" => {
            let mut formatter =
                select_formatter(LogFormat::Pretty, ConsoleSessionMode::InteractiveDashboard);
            assert_eq!(formatter.kind(), "interactive_dashboard");
            (
                LogFormat::Pretty,
                formatter
                    .format(&event)
                    .expect("format canonical pretty/TUI lifecycle event"),
            )
        }
        unsupported => panic!("unsupported terminal-boundary child mode: {unsupported}"),
    };

    write_rendered_output(format, &rendered).expect("write canonical lifecycle event to terminal");
}

#[test]
fn canonical_terminal_events_render_once_and_use_existing_event_controls() {
    let request_id = RequestId::new();
    let mut formatter = InteractiveDashboardFormatter::default();
    formatter
        .state
        .reduce(DashboardAction::Resize(DashboardLayoutState::new(
            3, 2, 2, 2, 2,
        )));

    for sequence in [1, 2] {
        formatter
            .handle_output_event(&canonical_output(
                request_id,
                sequence,
                LifecycleEvent::Completed {
                    status_code: Some(200),
                    duration_ms: Some(31),
                    usage: None,
                },
            ))
            .expect("canonical event should reduce");
    }
    for sequence in 3..10 {
        formatter
            .handle_output_event(&canonical_output(
                RequestId::new(),
                sequence,
                LifecycleEvent::Failed {
                    error: "private failure detail".to_string(),
                    status_code: None,
                },
            ))
            .expect("canonical event should reduce");
    }

    assert_eq!(formatter.state.mesh_events.len(), 8);
    assert_eq!(
        formatter
            .state
            .mesh_events
            .iter()
            .filter(|row| {
                row.summary
                    .starts_with("request completed status=200 duration=31ms request_id=")
            })
            .count(),
        1
    );
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('/')));
    for ch in "failed".chars() {
        formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char(ch)));
    }
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Enter));
    assert_eq!(formatter.state.filtered_mesh_events().len(), 7);
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::PageUp));
    assert!(!formatter.state.events_follow);
    formatter.handle_tui_event(TuiEvent::Key(TuiKeyEvent::Char('G')));
    assert!(formatter.state.events_follow);
}

#[test]
fn canonical_projection_keeps_local_correlation_but_redacts_payloads_and_native_ids() {
    let home = std::env::var("HOME").expect("HOME should be set");
    let request_id = RequestId::new();
    let request_id_text = request_id.as_uuid().to_string();
    let canonical = canonical_output(
        request_id,
        9,
        LifecycleEvent::Failed {
            error: format!(
                "prompt=private Bearer secret-token {home} completion=private response=private"
            ),
            status_code: None,
        },
    );
    let native = OutputEvent::LlamaNativeLog {
        message: format!("prompt=private {home} {}", "x".repeat(2_000)),
        category: "model",
        params: vec![
            (
                "request_id".to_string(),
                Value::String(request_id_text.clone()),
            ),
            ("private_path".to_string(), Value::String(home.clone())),
            ("progress".to_string(), serde_json::json!(50)),
        ],
    };
    let mut json = JsonFormatter;
    let mut pretty = PrettyFormatter;

    let canonical_json = json.format(&canonical).expect("canonical JSON projection");
    let canonical_pretty = pretty
        .format(&canonical)
        .expect("canonical pretty projection");
    for forbidden in ["private", "secret-token", home.as_str()] {
        assert!(
            !canonical_json.contains(forbidden),
            "canonical JSON leaked {forbidden:?}"
        );
        assert!(
            !canonical_pretty.contains(forbidden),
            "canonical pretty leaked {forbidden:?}"
        );
    }
    assert!(canonical_json.contains(&request_id_text));
    assert!(canonical_pretty.contains(&request_id_text));

    for event in [&native] {
        let json_line = json.format(event).expect("native JSON projection");
        let pretty_line = pretty.format(event).expect("native pretty projection");
        for forbidden in [
            "private",
            "secret-token",
            home.as_str(),
            request_id_text.as_str(),
            "tokens",
        ] {
            assert!(!json_line.contains(forbidden), "JSON leaked {forbidden:?}");
            assert!(
                !pretty_line.contains(forbidden),
                "pretty leaked {forbidden:?}"
            );
        }
        assert!(pretty_line.chars().count() <= 1_100);
    }
}

#[test]
fn canonical_stream_token_counts_are_local_operational_metadata() {
    let request_id = RequestId::new();
    let event = canonical_output(
        request_id,
        8,
        LifecycleEvent::StreamCompleted {
            tokens: Some(42),
            usage: Some(TokenUsage {
                prompt_tokens: Some(20),
                cached_prompt_tokens: None,
                completion_tokens: Some(42),
                total_tokens: Some(62),
            }),
        },
    );
    let json = JsonFormatter.format(&event).expect("canonical JSONL");
    let pretty = PrettyFormatter.format(&event).expect("canonical pretty");

    let record: Value = serde_json::from_str(json.trim_end()).expect("JSONL record");
    assert_eq!(record["tokens"], 42);
    assert_eq!(record["prompt_tokens"], 20);
    assert_eq!(record["completion_tokens"], 42);
    assert_eq!(record["total_tokens"], 62);
    assert_eq!(record["request_id"], request_id.as_uuid().to_string());
    assert!(pretty.contains("tokens=42"));
    assert!(pretty.contains(&request_id.as_uuid().to_string()));
}

#[test]
fn long_canonical_event_rows_wrap_to_the_event_panel_width() {
    let event = MeshEventState {
        timestamp: "12:34:56".to_string(),
        level: OutputLevel::Warn,
        summary: format!("request failed {}", "word ".repeat(200)),
    };

    let lines = wrapped_event_lines(&event, 48);
    assert!(lines.len() > 1);
    assert!(
        lines
            .iter()
            .all(|line| spans_plain_text(&line.spans).chars().count() <= 48)
    );
}
