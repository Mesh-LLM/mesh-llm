use super::*;
use crate::{clear_output_sink, set_output_sink};
use std::sync::{Arc, Mutex};

struct RecordingSink {
    mode: LogFormat,
    events: Mutex<Vec<OutputEvent>>,
}

impl RecordingSink {
    fn new(mode: LogFormat) -> Self {
        Self {
            mode,
            events: Mutex::new(Vec::new()),
        }
    }
}

impl OutputSink for RecordingSink {
    fn emit_event(&self, event: OutputEvent) -> io::Result<()> {
        self.events.lock().expect("recording sink lock").push(event);
        Ok(())
    }

    fn mode(&self) -> LogFormat {
        self.mode
    }
}

struct VerboseResetGuard;

impl Drop for VerboseResetGuard {
    fn drop(&mut self) {
        set_cli_command_event_verbose(false);
    }
}

struct OutputSinkResetGuard;

impl Drop for OutputSinkResetGuard {
    fn drop(&mut self) {
        clear_output_sink();
    }
}

#[test]
fn public_emit_is_silent_unless_verbose_enabled() {
    let sink = Arc::new(RecordingSink::new(LogFormat::Pretty));
    let _sink_guard = OutputSinkResetGuard;
    let _verbose_guard = VerboseResetGuard;
    set_output_sink(sink.clone());

    set_cli_command_event_verbose(false);
    emit_cli_command_event(CliCommandFamily::Runtime, CliCommandOutcome::Started)
        .expect("silent emission must succeed");
    assert!(
        sink.events.lock().expect("recording sink lock").is_empty(),
        "without --debug the command event must not be presented"
    );

    set_cli_command_event_verbose(true);
    emit_cli_command_event(CliCommandFamily::Runtime, CliCommandOutcome::Completed)
        .expect("verbose emission must succeed");
    assert_eq!(
        *sink.events.lock().expect("recording sink lock"),
        vec![OutputEvent::CliCommandLifecycle {
            family: CliCommandFamily::Runtime,
            outcome: CliCommandOutcome::Completed,
        }],
        "with --debug the command event must reach the pretty sink"
    );
}

#[test]
fn pretty_sink_receives_typed_command_event() {
    let sink = RecordingSink::new(LogFormat::Pretty);
    let mut stderr = Vec::new();
    let event = OutputEvent::CliCommandLifecycle {
        family: CliCommandFamily::Runtime,
        outcome: CliCommandOutcome::Started,
    };

    emit_cli_command_event_with_sink(&event, Some(&sink), &mut stderr)
        .expect("pretty sink should receive command event");

    assert_eq!(
        *sink.events.lock().expect("recording sink lock"),
        vec![event]
    );
    assert!(stderr.is_empty());
}

#[test]
fn json_sink_keeps_command_event_off_stdout() {
    let sink = RecordingSink::new(LogFormat::Json);
    let mut stderr = Vec::new();
    let event = OutputEvent::CliCommandLifecycle {
        family: CliCommandFamily::Models,
        outcome: CliCommandOutcome::Completed,
    };

    emit_cli_command_event_with_sink(&event, Some(&sink), &mut stderr)
        .expect("stderr fallback should write command event");

    assert!(sink.events.lock().expect("recording sink lock").is_empty());
    assert_eq!(
        String::from_utf8(stderr).expect("stderr must be utf-8"),
        "mesh-llm command event: family=models code=cli_command_completed outcome=completed\n"
    );
}

#[test]
fn command_event_vocabulary_is_bounded_and_static() {
    let families = [
        CliCommandFamily::Agent,
        CliCommandFamily::Benchmark,
        CliCommandFamily::Configuration,
        CliCommandFamily::Diagnostics,
        CliCommandFamily::Discovery,
        CliCommandFamily::Hardware,
        CliCommandFamily::Identity,
        CliCommandFamily::Installation,
        CliCommandFamily::Models,
        CliCommandFamily::Plugin,
        CliCommandFamily::Process,
        CliCommandFamily::Runtime,
        CliCommandFamily::Skills,
        CliCommandFamily::Unknown,
    ];
    let outcomes = [
        CliCommandOutcome::Started,
        CliCommandOutcome::Completed,
        CliCommandOutcome::Failed,
        CliCommandOutcome::Rejected,
        CliCommandOutcome::ParseFailed,
    ];

    for family in families {
        assert!(family.as_str().len() <= 24);
        assert!(
            family
                .as_str()
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
        );
    }
    for outcome in outcomes {
        assert!(outcome.code().len() <= 48);
        assert!(
            outcome
                .code()
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte == b'_')
        );
    }
}

#[test]
fn command_summary_rejects_control_text_and_unbounded_tokens() {
    assert!(CliCommandSummary::new("   ").is_none());
    assert!(CliCommandSummary::new("runtime\nload").is_none());
    assert!(CliCommandSummary::new(&"token ".repeat(33)).is_none());
    assert!(CliCommandSummary::new(&"x".repeat(257)).is_none());
}

#[test]
fn command_summary_sanitizer_rejects_arbitrary_values_and_preserves_safe_shape() {
    assert!(CliCommandSummary::sanitize("mesh-llm load model-private").is_none());
    assert!(CliCommandSummary::sanitize("mesh-llm load --port 41731 name [REDACTED]").is_some());
    assert!(CliCommandSummary::sanitize("mesh-llm load name [REDACTED]").is_some());
    assert!(CliCommandSummary::sanitize(&format!("mesh-llm {}", "load ".repeat(32))).is_none());
}

#[test]
fn command_summary_sanitizer_rejects_unknown_options_and_global_numeric_values() {
    assert!(CliCommandSummary::sanitize("mesh-llm --api-key 123456").is_none());
    assert!(CliCommandSummary::sanitize("mesh-llm load 123456 name [REDACTED]").is_none());
    assert!(CliCommandSummary::sanitize("mesh-llm load --port 41731 name [REDACTED]").is_some());
    assert!(
        CliCommandSummary::sanitize("mesh-llm runtime install runtime_ref [REDACTED]").is_some()
    );
}

#[test]
fn command_summary_sanitizer_rejects_non_canonical_whole_command_shapes() {
    let summaries = [
        "   ",
        "mesh-llm models list --json --json",
        "mesh-llm models --json list",
        "mesh-llm load name [REDACTED] name [REDACTED]",
        "mesh-llm gpus run-benchmark --backend cuda --json --json",
        "mesh-llm load --port 41731 --port 41732",
        "mesh-llm load --port 41731 name [REDACTED] --json",
        "mesh-llm load name [REDACTED] status",
        "mesh-llm models nonsense",
        "mesh-llm load --json name [REDACTED]",
        "mesh-llm runtime status name [REDACTED]",
        "mesh-llm models list name [REDACTED] --json",
    ];

    for summary in summaries {
        assert!(
            CliCommandSummary::sanitize(summary).is_none(),
            "non-canonical summary was accepted: {summary:?}"
        );
    }
}

#[test]
fn command_summary_sanitizer_rejects_non_canonical_ascii_whitespace() {
    for summary in [
        " mesh-llm models list",
        "mesh-llm models list ",
        "mesh-llm  models list",
        "mesh-llm\tmodels list",
        "mesh-llm models\nlist",
    ] {
        assert!(
            CliCommandSummary::sanitize(summary).is_none(),
            "non-canonical whitespace was accepted: {summary:?}"
        );
    }
}

#[test]
fn command_summary_sanitizer_rejects_conflicting_boolean_pairs() {
    for summary in [
        "mesh-llm setup --service --no-service",
        "mesh-llm setup --no-service --service",
        "mesh-llm uninstall --purge-config --keep-config",
        "mesh-llm uninstall --keep-config --purge-config",
        "mesh-llm auth init --no-passphrase --keychain",
        "mesh-llm auth init --keychain --no-passphrase",
    ] {
        assert!(
            CliCommandSummary::sanitize(summary).is_none(),
            "conflicting flags were accepted: {summary:?}"
        );
    }
}

#[test]
fn command_summary_sanitizer_accepts_global_relay_redaction_only() {
    assert!(
        CliCommandSummary::sanitize("mesh-llm load name [REDACTED] --root-relay [REDACTED]")
            .is_some()
    );
    for summary in [
        "mesh-llm load name [REDACTED] --relay private-relay",
        "mesh-llm load name [REDACTED] --root-relay [REDACTED] value",
        "mesh-llm load name [REDACTED] --relay-auth private-token",
        "mesh-llm load --root-relay [REDACTED] name [REDACTED]",
        "mesh-llm load name [REDACTED] --relay-auth [REDACTED] --root-relay [REDACTED]",
    ] {
        assert!(
            CliCommandSummary::sanitize(summary).is_none(),
            "malformed global relay marker was accepted: {summary:?}"
        );
    }
}

#[test]
fn command_summary_sanitizer_contextualizes_enum_values() {
    assert!(CliCommandSummary::sanitize("mesh-llm gpus run-benchmark --backend cuda").is_some());
    assert!(CliCommandSummary::sanitize("mesh-llm runtime guardrails --mode metrics").is_some());
    assert!(CliCommandSummary::sanitize("mesh-llm gpus run-benchmark --backend rocm").is_none());
    assert!(CliCommandSummary::sanitize("mesh-llm runtime guardrails --mode strict").is_none());
}

#[test]
fn command_summary_sanitizer_rejects_raw_options_in_wrong_command_contexts() {
    assert!(CliCommandSummary::sanitize("mesh-llm load --mode enforce").is_none());
    assert!(CliCommandSummary::sanitize("mesh-llm runtime status --backend cuda").is_none());
    assert!(CliCommandSummary::sanitize("mesh-llm gpus run-benchmark --port 1234").is_none());
}

#[test]
fn command_summary_sanitizer_rejects_deep_static_prefix_without_panicking() {
    let result = std::panic::catch_unwind(|| {
        CliCommandSummary::sanitize(
            "mesh-llm load unload status discover rotate-key setup --port 1234",
        )
    });

    assert!(result.is_ok());
    assert!(result.is_ok_and(|summary| summary.is_none()));
}

#[test]
fn command_summary_sanitizer_rejects_inserted_prefix_tokens_before_raw_options() {
    let summaries = [
        "mesh-llm gpus --draft run-benchmark --backend cuda",
        "mesh-llm gpus run-benchmark model [REDACTED] --backend cuda",
        "mesh-llm gpus run-benchmark detect --backend cuda",
        "mesh-llm runtime guardrails status --mode metrics",
        "mesh-llm gpus --json run-benchmark --backend cuda",
        "mesh-llm doctor --json split --port 1234",
    ];

    for summary in summaries {
        assert!(
            CliCommandSummary::sanitize(summary).is_none(),
            "malformed producer prefix was accepted: {summary}"
        );
    }
}

#[test]
fn command_summary_sanitizer_rejects_impossible_boolean_and_raw_option_orders() {
    let summaries = [
        "mesh-llm runtime guardrails --mode metrics --port 41731 --json",
        "mesh-llm runtime bootstrap --port 41731 --json",
        "mesh-llm runtime remote --port 41731 --json --endpoint [REDACTED]",
        "mesh-llm runtime remote --endpoint [REDACTED] --json --port 41731",
    ];

    for summary in summaries {
        assert!(
            CliCommandSummary::sanitize(summary).is_none(),
            "impossible producer ordering was accepted: {summary}"
        );
    }
}

#[test]
fn command_summary_sanitizer_accepts_every_producer_raw_option_context() {
    let summaries = [
        "mesh-llm status --port 41731",
        "mesh-llm load --port 41731 name [REDACTED]",
        "mesh-llm unload --port 41731 name [REDACTED]",
        "mesh-llm goose --port 41731 --model [REDACTED]",
        "mesh-llm claude --port 41731 --model [REDACTED]",
        "mesh-llm doctor split --port 41731 --model-ref [REDACTED]",
        "mesh-llm doctor split --json --port 41731 --model-ref [REDACTED]",
        "mesh-llm doctor split --json --port 41731 --model-ref [REDACTED]",
        "mesh-llm gpus run-benchmark --backend cuda",
        "mesh-llm gpus run-benchmark --backend cuda --json",
        "mesh-llm runtime status --port 41731",
        "mesh-llm runtime load --port 41731 name [REDACTED]",
        "mesh-llm runtime unload --port 41731 name [REDACTED]",
        "mesh-llm runtime guardrails --mode metrics --port 41731",
        "mesh-llm runtime guardrails --mode metrics --json --port 41731",
        "mesh-llm runtime bootstrap --port 41731",
        "mesh-llm runtime bootstrap --json --port 41731",
        "mesh-llm runtime remote --port 41731 --endpoint [REDACTED]",
        "mesh-llm runtime remote --json --port 41731 --endpoint [REDACTED]",
        "mesh-llm runtime remote-model --port 41731 --endpoint [REDACTED] --model [REDACTED]",
        "mesh-llm runtime remote-model --json --port 41731 --endpoint [REDACTED] --model [REDACTED]",
        "mesh-llm runtime apply-config --port 41731 --endpoint [REDACTED] --expected-revision [REDACTED] --config [REDACTED]",
        "mesh-llm runtime apply-config --json --port 41731 --endpoint [REDACTED] --expected-revision [REDACTED] --config [REDACTED]",
    ];

    for summary in summaries {
        assert!(
            CliCommandSummary::sanitize(summary).is_some(),
            "producer-reachable summary was rejected: {summary}"
        );
    }
}
