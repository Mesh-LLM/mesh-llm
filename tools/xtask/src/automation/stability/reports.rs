use super::{cases::Case, release_attestation::ResultRow};
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(super) enum CommandStatus {
    Pass,
    Fail,
    Prereq,
}

impl CommandStatus {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::Prereq => "PREREQ",
        }
    }
}

#[derive(Serialize)]
pub(super) struct CommandRow {
    pub name: String,
    pub status: CommandStatus,
    pub exit_code: i32,
    pub elapsed_ms: u64,
    pub log: String,
    pub detail: String,
}

#[derive(Default, Debug, Serialize)]
pub(super) struct Counts {
    pub ok: bool,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub prereq: usize,
    pub elapsed_ms: u64,
}

impl Counts {
    fn add(&mut self, status: CommandStatus, elapsed_ms: u64) {
        self.total += 1;
        match status {
            CommandStatus::Pass => self.passed += 1,
            CommandStatus::Fail => self.failed += 1,
            CommandStatus::Prereq => self.prereq += 1,
        }
        self.elapsed_ms = self.elapsed_ms.saturating_add(elapsed_ms);
        self.ok = self.failed == 0;
    }

    fn combine(parts: &[&Self]) -> Self {
        let mut result = Self {
            ok: true,
            ..Self::default()
        };
        for part in parts {
            result.total += part.total;
            result.passed += part.passed;
            result.failed += part.failed;
            result.prereq += part.prereq;
            result.elapsed_ms = result.elapsed_ms.saturating_add(part.elapsed_ms);
            result.ok &= part.ok;
        }
        result
    }
}

#[derive(Serialize)]
pub(super) struct AttestationCounts {
    #[serde(flatten)]
    pub counts: Counts,
    pub status: String,
}

#[derive(Serialize)]
pub(super) struct Summary<'a> {
    #[serde(flatten)]
    pub counts: Counts,
    pub cancelled: bool,
    pub commands: Counts,
    pub probes: Counts,
    pub attestation: AttestationCounts,
    pub release_attestation: &'a ResultRow,
}

pub(super) fn summarize<'a>(
    commands: &[CommandRow],
    probes: &[Case],
    attestation: &'a ResultRow,
    cancelled: bool,
) -> Summary<'a> {
    let mut command_counts = Counts {
        ok: true,
        ..Counts::default()
    };
    for row in commands {
        command_counts.add(row.status, row.elapsed_ms);
    }
    let mut probe_counts = Counts {
        ok: true,
        ..Counts::default()
    };
    for row in probes {
        probe_counts.add(
            if row.ok {
                CommandStatus::Pass
            } else {
                CommandStatus::Fail
            },
            row.elapsed_ms,
        );
    }
    let mut attestation_counts = Counts::default();
    let status = if !attestation.ok {
        CommandStatus::Fail
    } else if attestation.status == "not_configured" {
        CommandStatus::Prereq
    } else {
        CommandStatus::Pass
    };
    attestation_counts.add(status, attestation.elapsed_ms);
    let mut counts = Counts::combine(&[&command_counts, &probe_counts, &attestation_counts]);
    counts.ok &= !cancelled;
    Summary {
        counts,
        cancelled,
        commands: command_counts,
        probes: probe_counts,
        attestation: AttestationCounts {
            counts: attestation_counts,
            status: attestation.status.clone(),
        },
        release_attestation: attestation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unconfigured() -> ResultRow {
        ResultRow {
            status: "not_configured".into(),
            ok: true,
            binary: None,
            expected_status: None,
            version: None,
            signer_key_id: None,
            artifact_digest: None,
            error: None,
            elapsed_ms: 0,
        }
    }

    #[test]
    fn absent_optional_attestation_is_explicit_and_cancellation_cannot_pass() {
        let attestation = unconfigured();
        let summary = summarize(&[], &[], &attestation, false);
        assert!(summary.counts.ok);
        assert_eq!(summary.counts.total, 1);
        assert_eq!(summary.counts.prereq, 1);
        assert_eq!(summary.counts.passed, 0);
        let partial = summarize(&[], &[], &attestation, true);
        assert!(!partial.counts.ok);
        assert!(partial.cancelled);
    }

    #[test]
    fn optional_prerequisites_and_real_command_failures_are_counted_separately() {
        let rows = [
            CommandRow {
                name: "pi".into(),
                status: CommandStatus::Prereq,
                exit_code: 0,
                elapsed_ms: 0,
                log: "logs/pi.log".into(),
                detail: "CLI missing".into(),
            },
            CommandRow {
                name: "tool-call-reliability".into(),
                status: CommandStatus::Fail,
                exit_code: 1,
                elapsed_ms: 13,
                log: "logs/tool.log".into(),
                detail: "wrong answer".into(),
            },
        ];
        let attestation = unconfigured();
        let summary = summarize(&rows, &[], &attestation, false);
        assert!(!summary.counts.ok);
        assert_eq!(summary.counts.total, 3);
        assert_eq!(summary.counts.failed, 1);
        assert_eq!(summary.counts.prereq, 2);
        assert_eq!(summary.counts.elapsed_ms, 13);
    }
}
