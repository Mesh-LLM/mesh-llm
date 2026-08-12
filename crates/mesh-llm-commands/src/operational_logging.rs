//! Bounded CLI command dispatch events.
//!
//! This boundary starts after `clap` has parsed a [`Command`] and finishes at
//! the top-level command dispatcher. It deliberately does not inspect command
//! arguments or error text. Parse failures happen before a `Command` exists
//! and remain outside this boundary.

use anyhow::Result;
use mesh_llm_cli::Command;
#[cfg(test)]
use mesh_llm_events::OutputEvent;
use mesh_llm_events::{CliCommandFamily, CliCommandOutcome, emit_cli_command_event};
use std::fmt;
use std::sync::{Arc, LazyLock, RwLock};

/// Marker a handler can retain in its error chain when it explicitly rejects a
/// parsed command. Other handler or dispatcher errors are classified as
/// failures without reading their details.
#[derive(Debug)]
pub struct CommandDispatchRejected;

impl fmt::Display for CommandDispatchRejected {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("command dispatch rejected")
    }
}

impl std::error::Error for CommandDispatchRejected {}

/// Fail-open durable-audit bridge invoked after each command boundary
/// emission. The caller installs it only after logging runtime
/// initialization, so no durable record is produced before initialization.
/// The bridge receives only the static command family and outcome; it never
/// sees command arguments, argv, or error text.
pub type CliOperationalAuditBridge = Arc<dyn Fn(CliCommandFamily, CliCommandOutcome) + Send + Sync>;

static CLI_OPERATIONAL_AUDIT_BRIDGE: LazyLock<RwLock<Option<CliOperationalAuditBridge>>> =
    LazyLock::new(|| RwLock::new(None));

/// Install the durable command audit bridge. Callers invoke this only after
/// the logging runtime has been initialized so command behavior is unchanged
/// before initialization. Replacing the bridge is allowed for tests and
/// embedded hosts.
pub fn install_cli_operational_audit_bridge(bridge: CliOperationalAuditBridge) {
    *CLI_OPERATIONAL_AUDIT_BRIDGE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(bridge);
}

/// Remove the command audit bridge. Used by tests and by callers that must
/// stop persisting command outcomes.
pub fn clear_cli_operational_audit_bridge() {
    *CLI_OPERATIONAL_AUDIT_BRIDGE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
}

/// Return the installed bridge, if any.
pub fn cli_operational_audit_bridge() -> Option<CliOperationalAuditBridge> {
    CLI_OPERATIONAL_AUDIT_BRIDGE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clone()
}

/// Process-local lifecycle emitter for one parsed command dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommandDispatchBoundary {
    family: CliCommandFamily,
}

impl CommandDispatchBoundary {
    /// Start the command boundary after parsing succeeds.
    pub fn start(command: &Command) -> Self {
        let boundary = Self {
            family: command_family(command),
        };
        boundary.emit(CliCommandOutcome::Started);
        boundary
    }

    /// Emit exactly one terminal result for the command dispatcher outcome.
    pub fn finish(self, result: &Result<()>) {
        self.emit(command_outcome(result));
    }

    fn emit(self, outcome: CliCommandOutcome) {
        // Command results must retain their existing return and output behavior
        // even if the process-local presentation sink is unavailable.
        let _ = emit_cli_command_event(self.family, outcome);
        if let Some(bridge) = cli_operational_audit_bridge() {
            bridge(self.family, outcome);
        }
    }

    #[cfg(test)]
    fn event(self, outcome: CliCommandOutcome) -> OutputEvent {
        OutputEvent::CliCommandLifecycle {
            family: self.family,
            outcome,
        }
    }
}

/// Map a parsed command to a stable, argument-free event family.
pub fn command_family(command: &Command) -> CliCommandFamily {
    match command {
        Command::Models { .. } | Command::Download { .. } | Command::ModelPrepare { .. } => {
            CliCommandFamily::Models
        }
        Command::Update { .. } | Command::Setup { .. } | Command::Uninstall { .. } => {
            CliCommandFamily::Installation
        }
        Command::Gpus { .. } => CliCommandFamily::Hardware,
        Command::Runtime { .. }
        | Command::Load { .. }
        | Command::Unload { .. }
        | Command::Status { .. }
        | Command::Stop => CliCommandFamily::Runtime,
        Command::Config { .. } => CliCommandFamily::Configuration,
        Command::Doctor { .. } => CliCommandFamily::Diagnostics,
        Command::Discover { .. } => CliCommandFamily::Discovery,
        Command::RotateKey | Command::Auth { .. } => CliCommandFamily::Identity,
        Command::Goose { .. }
        | Command::Claude { .. }
        | Command::Pi { .. }
        | Command::Opencode { .. } => CliCommandFamily::Agent,
        Command::Plugin { .. } | Command::ExternalPlugin(_) => CliCommandFamily::Plugin,
        Command::Skills { .. } => CliCommandFamily::Skills,
        Command::Benchmark { .. } => CliCommandFamily::Benchmark,
    }
}

fn command_outcome(result: &Result<()>) -> CliCommandOutcome {
    match result {
        Ok(()) => CliCommandOutcome::Completed,
        Err(error) if error.downcast_ref::<CommandDispatchRejected>().is_some() => {
            CliCommandOutcome::Rejected
        }
        Err(_) => CliCommandOutcome::Failed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Error, anyhow};
    use mesh_llm_events::{clear_output_sink, set_output_sink};
    use std::io;

    #[derive(Default)]
    struct RecordingBridge {
        calls: std::sync::Mutex<Vec<(CliCommandFamily, CliCommandOutcome)>>,
    }

    impl RecordingBridge {
        fn take_calls(&self) -> Vec<(CliCommandFamily, CliCommandOutcome)> {
            std::mem::take(&mut *self.calls.lock().expect("recording bridge mutex poisoned"))
        }
    }

    struct BridgeResetGuard;

    impl Drop for BridgeResetGuard {
        fn drop(&mut self) {
            clear_cli_operational_audit_bridge();
        }
    }

    #[derive(Default)]
    struct RecordingOutputSink {
        events: std::sync::Mutex<Vec<OutputEvent>>,
    }

    impl RecordingOutputSink {
        fn take_events(&self) -> Vec<OutputEvent> {
            std::mem::take(&mut *self.events.lock().expect("recording sink mutex poisoned"))
        }
    }

    impl mesh_llm_events::OutputSink for RecordingOutputSink {
        fn emit_event(&self, event: OutputEvent) -> io::Result<()> {
            self.events
                .lock()
                .expect("recording sink mutex poisoned")
                .push(event);
            Ok(())
        }
    }

    struct OutputSinkResetGuard;

    impl Drop for OutputSinkResetGuard {
        fn drop(&mut self) {
            clear_output_sink();
        }
    }

    fn lifecycle_events(command: &Command, result: &Result<()>) -> [OutputEvent; 2] {
        let boundary = CommandDispatchBoundary {
            family: command_family(command),
        };
        [
            boundary.event(CliCommandOutcome::Started),
            boundary.event(command_outcome(result)),
        ]
    }

    #[test]
    fn command_dispatch_orders_started_before_completed_without_command_arguments() {
        let command = Command::Load {
            name: "private-model.gguf?token=private-token".to_string(),
            port: 41731,
        };
        let result = Ok(());

        let events = lifecycle_events(&command, &result);

        assert_eq!(
            events,
            [
                OutputEvent::CliCommandLifecycle {
                    family: CliCommandFamily::Runtime,
                    outcome: CliCommandOutcome::Started,
                },
                OutputEvent::CliCommandLifecycle {
                    family: CliCommandFamily::Runtime,
                    outcome: CliCommandOutcome::Completed,
                },
            ]
        );
        let serialized = format!("{events:?}");
        for raw_value in [
            "private-model.gguf?token=private-token",
            "41731",
            "private-token",
        ] {
            assert!(
                !serialized.contains(raw_value),
                "command metadata must not enter lifecycle events"
            );
        }
    }

    #[test]
    fn explicit_handler_rejection_maps_to_rejected_without_error_detail() {
        let command = Command::Discover {
            name: Some("private-mesh".to_string()),
            model: None,
            min_vram: None,
            region: Some("private-region".to_string()),
            auto: false,
            relay: vec!["wss://relay.private.example/?token=private-token".to_string()],
        };
        let result: Result<()> = Err(Error::new(CommandDispatchRejected)
            .context("private command rejection detail with private-token"));

        let events = lifecycle_events(&command, &result);

        assert_eq!(
            events[1],
            OutputEvent::CliCommandLifecycle {
                family: CliCommandFamily::Discovery,
                outcome: CliCommandOutcome::Rejected,
            }
        );
        let serialized = format!("{events:?}");
        for raw_value in [
            "private-mesh",
            "private-region",
            "wss://relay.private.example/?token=private-token",
            "private command rejection detail with private-token",
        ] {
            assert!(
                !serialized.contains(raw_value),
                "rejection detail must not enter lifecycle events"
            );
        }
    }

    #[test]
    fn unmarked_dispatch_failure_maps_to_failed_without_error_detail() {
        let command = Command::Download {
            name: Some("private/model?token=private-token".to_string()),
            draft: true,
        };
        let result: Result<()> = Err(anyhow!(
            "download failed for https://private.example/model?token=private-token"
        ));

        let events = lifecycle_events(&command, &result);

        assert_eq!(
            events[1],
            OutputEvent::CliCommandLifecycle {
                family: CliCommandFamily::Models,
                outcome: CliCommandOutcome::Failed,
            }
        );
        let serialized = format!("{events:?}");
        assert!(!serialized.contains("https://private.example/model?token=private-token"));
    }

    #[test]
    #[serial_test::serial]
    fn command_dispatch_keeps_output_events_unchanged_when_bridge_installed() {
        let sink = Arc::new(RecordingOutputSink::default());
        let _sink_guard = OutputSinkResetGuard;
        set_output_sink(sink.clone());
        let recording = Arc::new(RecordingBridge::default());
        let _bridge_guard = BridgeResetGuard;
        let bridge_recording = recording.clone();
        install_cli_operational_audit_bridge(Arc::new(move |family, outcome| {
            bridge_recording
                .calls
                .lock()
                .expect("recording bridge mutex poisoned")
                .push((family, outcome));
        }));

        let command = Command::Load {
            name: "model.gguf".to_string(),
            port: 9337,
        };
        let boundary = CommandDispatchBoundary::start(&command);
        let result: Result<()> = Ok(());
        boundary.finish(&result);

        assert_eq!(
            sink.take_events(),
            vec![
                OutputEvent::CliCommandLifecycle {
                    family: CliCommandFamily::Runtime,
                    outcome: CliCommandOutcome::Started,
                },
                OutputEvent::CliCommandLifecycle {
                    family: CliCommandFamily::Runtime,
                    outcome: CliCommandOutcome::Completed,
                },
            ]
        );
        assert_eq!(
            recording.take_calls(),
            vec![
                (CliCommandFamily::Runtime, CliCommandOutcome::Started),
                (CliCommandFamily::Runtime, CliCommandOutcome::Completed),
            ],
            "installed bridge fires exactly once per outcome"
        );
    }

    #[test]
    #[serial_test::serial]
    fn command_dispatch_without_bridge_stays_fail_open() {
        clear_cli_operational_audit_bridge();
        let command = Command::Load {
            name: "private-model.gguf".to_string(),
            port: 41731,
        };
        let boundary = CommandDispatchBoundary::start(&command);
        let result: Result<()> = Err(anyhow!("private dispatch failure"));
        boundary.finish(&result);

        assert!(
            cli_operational_audit_bridge().is_none(),
            "uninstalled bridge must remain absent after dispatch"
        );
    }

    #[test]
    #[serial_test::serial]
    fn bridge_receives_only_static_family_and_outcome_without_command_arguments() {
        let recording = Arc::new(RecordingBridge::default());
        let _bridge_guard = BridgeResetGuard;
        let bridge_recording = recording.clone();
        install_cli_operational_audit_bridge(Arc::new(move |family, outcome| {
            bridge_recording
                .calls
                .lock()
                .expect("recording bridge mutex poisoned")
                .push((family, outcome));
        }));

        let command = Command::Discover {
            name: Some("private-mesh".to_string()),
            model: None,
            min_vram: None,
            region: Some("private-region".to_string()),
            auto: false,
            relay: vec!["wss://relay.private.example/?token=private-token".to_string()],
        };
        let boundary = CommandDispatchBoundary::start(&command);
        let result: Result<()> = Err(Error::new(CommandDispatchRejected)
            .context("private command rejection detail with private-token"));
        boundary.finish(&result);

        let calls = recording.take_calls();
        assert_eq!(
            calls,
            vec![
                (CliCommandFamily::Discovery, CliCommandOutcome::Started),
                (CliCommandFamily::Discovery, CliCommandOutcome::Rejected),
            ]
        );
        let serialized = format!("{calls:?}");
        for raw_value in [
            "private-mesh",
            "private-region",
            "wss://relay.private.example/?token=private-token",
            "private command rejection detail with private-token",
        ] {
            assert!(
                !serialized.contains(raw_value),
                "command metadata must not cross the audit bridge"
            );
        }
    }
}
