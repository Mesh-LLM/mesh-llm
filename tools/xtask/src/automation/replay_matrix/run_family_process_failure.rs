use super::run_family_report::RunFamilyReport;
use crate::process::{Outcome, ProcessReport};

pub(super) fn success_output(
    print_shell: bool,
    shell_line: String,
    report: ProcessReport,
) -> RunFamilyReport {
    let mut stdout = report.stdout.bytes_retained;
    if print_shell {
        stdout.extend_from_slice(shell_line.as_bytes());
    }
    RunFamilyReport::success(stdout, retained_diagnostics(&report.stderr.bytes_retained))
}

pub(super) fn report(report: ProcessReport) -> RunFamilyReport {
    let status = report.status.and_then(|status| status.code());
    let outcome = match report.outcome {
        Outcome::Exited => "Exited",
        Outcome::Ready => "Ready",
        Outcome::EarlyExit => "EarlyExit",
        Outcome::Deadline => "Deadline",
        Outcome::ReadinessDeadline => "ReadinessDeadline",
        Outcome::Cancelled => "Cancelled",
        Outcome::IoFailure => "IoFailure",
        Outcome::ObservationRejected => "ObservationRejected",
    };
    RunFamilyReport::failed_child(
        report.stdout.bytes_retained,
        format!(
            "run-family child failed: outcome={outcome}, status={status:?}, cleanup={:?}, failure={:?}\nstderr: {}",
            report.cleanup,
            report.failure,
            String::from_utf8_lossy(&retained_diagnostics(&report.stderr.bytes_retained))
        ),
    )
}

pub(super) fn interrupted(report: ProcessReport) -> RunFamilyReport {
    RunFamilyReport::failure(format!(
        "run-family cancelled; outcome={:?}, status={:?}, cleanup={:?}\n",
        report.outcome, report.status, report.cleanup
    ))
}

fn retained_diagnostics(bytes: &[u8]) -> Vec<u8> {
    bytes
        .split_inclusive(|byte| *byte == b'\n')
        .filter(|line| crate::automation::replay_matrix::progress::decode(line).is_none())
        .flatten()
        .copied()
        .collect()
}
