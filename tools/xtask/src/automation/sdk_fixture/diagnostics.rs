use crate::process::retained::{MemberId, Report};

/// Report process facts only: no captured output, rejection text, paths or secrets.
pub(super) fn summary(report: &Report<String>) -> String {
    let mut parts = vec![format!(
        "outcome={:?}, rejected={}, supervision_failed={}",
        report.outcome,
        report.rejection.is_some(),
        report.failure.is_some()
    )];
    for member in &report.members {
        let role = if member.member == MemberId::Seed {
            "daemon"
        } else if member.member == MemberId::WorkerOne {
            "consumer"
        } else {
            "runtime-install"
        };
        let process = &member.process;
        let status = process
            .status
            .map_or_else(|| "unobserved".to_owned(), |status| status.to_string());
        parts.push(format!(
            "{role}: disposition={}, outcome={:?}, exit_code={:?}, status={status}, elapsed_ms={}, cleanup_complete={}, forced={}, signal_failed={}, cleanup_failed={}, process_failed={}",
            member.disposition.label(),
            process.outcome,
            process.status.and_then(|status| status.code()),
            process.elapsed.as_millis(),
            process.cleanup.complete,
            process.cleanup.forced,
            process.cleanup.graceful_signal_failed,
            process.cleanup.failure.is_some(),
            process.failure.is_some()
        ));
    }
    parts.join("; ")
}
